from django.db import models
from django.utils.text import slugify

from .models_dataset import Dataset as DatasetModel
from .models_setLabel import SetLabel as SetLabelModel
from .models_setFitur import SetFitur as SetFiturModel
from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel


def validate_k_cv(value):
    if int(value) <= 2:
        raise ValidationError(
            u'K-FOLDS CV : mohon maaf,nilai "K" dalam K-Folds Cross Validation harus lebih besar atau sama dengan 2')


class RandomForest(models.Model):
    dataset = models.ForeignKey(
        DatasetModel, on_delete=models.CASCADE)
    setlabel = models.ForeignKey(
        SetLabelModel, on_delete=models.CASCADE)
    setfitur = models.ForeignKey(
        SetFiturModel, on_delete=models.CASCADE)
    hyperparameter = models.ForeignKey(
        HyperparameterRFModel, on_delete=models.CASCADE)
    k_cv = models.CharField(max_length=255, default=5,
                            validators=[validate_k_cv])
    rf_result = models.FileField(
        upload_to='randomForest/result/', default='coba.csv')
    rf_fitur_importance = models.FileField(
        upload_to='randomForest/fiturImportance/', default='coba.csv')
    rf_model = models.FileField(
        upload_to='randomForest/model/', default='coba.pkl')

    tanggal_running = models.DateTimeField(auto_now_add=True)

    def delete(self, *args, **kwargs):
        self.rf_result.delete()
        self.rf_fitur_importance.delete()
        self.rf_model.delete()
        super().delete(*args, **kwargs)

    def __str__(self):
        return "[{}] 'RF-'{}".format(self.id, str(self.tanggal_running)[1:18])
