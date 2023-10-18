'''
This is a model class. Model: Bert with adapter + MLP Classifier
'''

from transformers import AutoAdapterModel
import torch.nn as nn

class DisasterClassifier(nn.Module):
    def __init__(self, bert_out, adapter, adapter_config, dime_reduction_layer_out, hidden_layer_1, hidden_layer_2, droup_out_rate):
        super(DisasterClassifier, self).__init__()
        
        self.bert_out = bert_out
        self.adapter = adapter
        self.adapter_config = adapter_config
        self.dime_reduction_layer_out = dime_reduction_layer_out
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.droup_out_rate = droup_out_rate

        self.bert = AutoAdapterModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        # Load pre-trained task adapter from Adapter Hub
        # This method call will also load a pre-trained classification head for the adapter task  
        
        # https://github.com/adapter-hub/adapter-transformers/blob/master/adapter_docs/prediction_heads.md
 
        self.bert.add_classification_head(self.adapter, num_labels=self.bert_out)
        
        self.bert.add_adapter(self.adapter, config=self.adapter_config)
        self.bert.set_active_adapters(self.adapter)
        
        self.dimension_reduce_layer = nn.Linear(self.bert_out, self.dime_reduction_layer_out)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dime_reduction_layer_out, self.hidden_layer_1),
            nn.ReLU(),
            nn.Dropout(self.droup_out_rate),
            nn.Linear(self.hidden_layer_1, self.hidden_layer_2),
            nn.ReLU(),
            nn.Dropout(self.droup_out_rate),
            nn.Linear(self.hidden_layer_2, 2),  
        )

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs['logits']
        pooled_output = self.dimension_reduce_layer(pooled_output)
        logits = self.mlp(pooled_output)
        return logits