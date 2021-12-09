from tensorflow.keras.models import Sequential
import numpy as np


class DANN():
    def __init__(self, feature_extractor, label_predictor, domain_classifier, loss_lambda=1.):
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier
        
        self.loss_lambda = loss_lambda
        
        self.label_model_ = Sequential([feature_extractor, label_predictor])
        self.domain_model_d_ = Sequential([feature_extractor, domain_classifier])
        self.domain_model_f_ = Sequential([feature_extractor, domain_classifier])
        
        self.label_model_.compile(
            optimizer="Adam",
            loss='binary_crossentropy', 
            loss_weights=[1.], 
            metrics=['accuracy']
        )
        
        feature_extractor.trainable = False
        self.domain_model_d_.compile(
            optimizer="Adam",
            loss='binary_crossentropy', 
            loss_weights=[self.loss_lambda], 
            metrics=['accuracy']
        )
        
        feature_extractor.trainable = True
        domain_classifier.trainable = False
        self.domain_model_f_.compile(
            optimizer="Adam",
            loss='binary_crossentropy', 
            loss_weights=[self.loss_lambda], 
            metrics=['accuracy']
        )
        
    def __call__(self, *args, **kwargs):
        return self.label_model_(*args, **kwargs).numpy()
    
    def draw_batch(self, X_source, X_target, y_source, batch_size):
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        source_indices = np.random.choice(n_source, size=batch_size, replace=False)
        target_indices = np.random.choice(n_target, size=batch_size, replace=False)
        
        X_mixed_ = np.concatenate([X_source[source_indices], X_target[target_indices]])
        y_domain_ = np.array([0] * batch_size + [1] * batch_size)
        y_label_ = y_source[source_indices]
        
        return X_mixed_, y_domain_, y_label_
        
    def fit(self, tup_X, y, epochs=1, batch_size=32, verbose=2):
        X_source, X_target = tup_X
        n_batches = X_source.shape[0] // batch_size

        for ep in range(1, epochs + 1):
            for b in range(n_batches):
                X_mixed, y_domain, y_label = self.draw_batch(
                    X_source, X_target, y, batch_size
                )
                
                self.label_model_.train_on_batch(
                    X_mixed[:batch_size], y_label,
                    sample_weight=2 * np.ones((batch_size, ))
                )
                
                self.domain_model_f_.train_on_batch(X_mixed, 1 - y_domain)                
                self.domain_model_d_.train_on_batch(X_mixed, y_domain)
                
                