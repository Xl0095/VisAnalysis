from transformers import BertForSequenceClassification
from torch.optim import AdamW

def load_model_optimizer(model_name, lr):
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        model_name, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        return_dict = False, # Or loss and logits will be a 'str' type
    )

    optimizer = AdamW(model.parameters(),
                lr = lr, # args.learning_rate - default is 5e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )
    
    return model, optimizer