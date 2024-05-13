from classif_model_builder import build_classif_model_on_full_train

HF_MODEL_LIST = {
    'en': [
           'google/electra-base-discriminator',
          ],
    'es': [
            'PlanTL-GOB-ES/roberta-base-bne',
          ],
}

DEFAULT_RND_SEED = 564671

# Train and save the English model
english_model = build_classif_model_on_full_train(
    lang='en',
    model_name=HF_MODEL_LIST['en'],
    model_label='electra-en',
    rseed=DEFAULT_RND_SEED
)

# Train and save the Spanish model
spanish_model = build_classif_model_on_full_train(
    lang='es',
    model_name=HF_MODEL_LIST['es'],
    model_label='roberta-bne-es',
    rseed=DEFAULT_RND_SEED
)