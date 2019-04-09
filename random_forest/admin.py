from django.contrib import admin

from .models_dataset import Dataset
from .models_setLabel import SetLabel
from .models_setFitur import SetFitur
from .models_hyperparameterRF import HyperparameterRF
from .models_randomForest import RandomForest

# Register your models here.
admin.site.register([Dataset, SetLabel, SetFitur,
                     HyperparameterRF, RandomForest])
