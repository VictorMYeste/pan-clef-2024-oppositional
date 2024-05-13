from classif_experim import classif_model_builder

DEFAULT_RND_SEED = 564671

# Train and save the English model
english_model = classif_model_builder.build_classif_model_on_full_train(
    lang='en',
    model_name='google/electra-base-discriminator',
    model_label='electra-en',
    rseed=DEFAULT_RND_SEED
)

# Train and save the Spanish model
spanish_model = classif_model_builder.build_classif_model_on_full_train(
    lang='es',
    model_name='PlanTL-GOB-ES/roberta-base-bne',
    model_label='roberta-bne-es',
    rseed=DEFAULT_RND_SEED
)