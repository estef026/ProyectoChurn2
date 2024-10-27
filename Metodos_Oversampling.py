from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from collections import Counter

def random_oversampling(X, y, random_state=42):
    """Aplica Random Oversampling a los datos."""
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print(f"Distribución de clases después de Random Oversampling: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def smote_oversampling(X, y, random_state=42):
    """Aplica SMOTE a los datos."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Distribución de clases después de SMOTE: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def adasyn_oversampling(X, y, random_state=42):
    """Aplica ADASYN a los datos."""
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    print(f"Distribución de clases después de ADASYN: {Counter(y_resampled)}")
    return X_resampled, y_resampled
