from tensorflow.keras.models import Sequential
import numpy as np


class DANN():
    def __init__(self, feature_extractor, label_predictor, domain_classifier, loss_lambda=1.):
        """
        Using the notations from the DANN paper [1],
        
        * `feature_extractor` is $G_f$
        * `label_predictor` is $G_y$
        * `domain_classifier` is $G_d$
        * `loss_lambda` is the hyper-parameter $\lambda$ in Eq. 10
        
        [1] https://arxiv.org/pdf/1505.07818.pdf
        """
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier
        
        self.loss_lambda = loss_lambda
        
        self.label_model_ = Sequential([feature_extractor, 
                                        label_predictor])
        self.domain_model_d_ = Sequential([feature_extractor, 
                                           domain_classifier])
        self.domain_model_f_ = Sequential([feature_extractor, 
                                           domain_classifier])
        self.epochs_ = 0
    
    def compile(self, optimizer, loss, metrics=[]):
        """
        The loss, optimizer and metrics provided here are those 
        to be used for the label predictor.
        """
        self.label_model_.compile(
            optimizer=optimizer,
            loss=loss, 
            loss_weights=[1.], 
            metrics=metrics
        )
        
        self.feature_extractor.trainable = False
        self.domain_model_d_.compile(
            optimizer="Adam",
            loss='binary_crossentropy', 
            loss_weights=[self.loss_lambda], 
            metrics=['accuracy']
        )
        
        self.feature_extractor.trainable = True
        self.domain_classifier.trainable = False
        self.domain_model_f_.compile(
            optimizer="Adam",
            loss='binary_crossentropy', 
            loss_weights=[self.loss_lambda], 
            metrics=['accuracy']
        )
        
    def __call__(self, *args, **kwargs):
        return self.label_model_(*args, **kwargs).numpy()
    
    def _batches(self, X_source, X_target, y_source, batch_size):
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        n_batches = n_source // batch_size
        for _ in range(n_batches):
            source_indices = np.random.choice(n_source, 
                                              size=batch_size, 
                                              replace=False)
            target_indices = np.random.choice(n_target, 
                                              size=batch_size, 
                                              replace=False)
            
            X_mixed_ = np.concatenate([X_source[source_indices], 
                                       X_target[target_indices]])
            y_domain_ = np.array([0] * batch_size + [1] * batch_size)
            y_label_ = y_source[source_indices]
            
            yield X_mixed_, y_domain_, y_label_
            
    def _str_stats(self, stats, preffix=""):
        if preffix != "" and not preffix.endswith("_"):
            preffix = preffix + "_"
        l = [f", {preffix}{k}: {v:.6f}" for k, v in stats.items()]
        return "".join(l)
        
    def fit(self, tup_X, y, epochs=1, batch_size=32, verbose=2):
        X_source, X_target = tup_X
        for ep in range(1, epochs + 1):
            for X_mixed, y_domain, y_label in self._batches(X_source, X_target, 
                                                            y, batch_size):
                source_stats = self.label_model_.train_on_batch(
                    X_mixed[:batch_size], y_label,
                    sample_weight=2 * np.ones((batch_size, )),
                    return_dict=True
                )                
                self.domain_model_f_.train_on_batch(X_mixed, 1 - y_domain)                
                domain_stats = self.domain_model_d_.train_on_batch(X_mixed, 
                                                                   y_domain,
                                                                   return_dict=True)
            if verbose > 0:
                print(f"Epoch {ep + self.epochs_}"
                      f"{self._str_stats(source_stats, 'source')}" 
                      f"{self._str_stats(domain_stats, 'domain')}")
        self.epochs_ += ep
                
                