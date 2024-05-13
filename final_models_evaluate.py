from hf_skelarn_wrapper import SklearnTransformerClassif
from classif_experiment_eval import evaluate_on_test_dataset
from classif_model_builder import get_model_folder_name

HF_MODEL_LIST = {
    'en': [
           'google/electra-base-discriminator',
          ],
    'es': [
            'PlanTL-GOB-ES/roberta-base-bne',
          ],
}

DEFAULT_RND_SEED = 564671

# Load the English model
english_model_path = get_model_folder_name('en', HF_MODEL_LIST['en'], DEFAULT_RND_SEED, 'conspiracy')
english_model = SklearnTransformerClassif.load(english_model_path)

# Load the Spanish model
spanish_model_path = get_model_folder_name('es', HF_MODEL_LIST['es'], DEFAULT_RND_SEED, 'conspiracy')
spanish_model = SklearnTransformerClassif.load(spanish_model_path)

# Evaluate on the official test datasets
evaluate_on_test_dataset(english_model, lang='en', positive_class='conspiracy')
evaluate_on_test_dataset(spanish_model, lang='es', positive_class='conspiracy')
