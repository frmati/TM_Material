
from transformers import BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
tokenizer_base = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

tokenizer_paraphrase = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
#model = BertModel.from_pretrained("bert-base-multilingual-uncased")


text = 'Raumschiffskapitän'

encoded_input = tokenizer_base.encode(text, add_special_tokens=False, add_prefix_space=True) #return_tensors='pt' 

print({x : tokenizer_base.encode(x, add_special_tokens=False, add_prefix_space=True) for x in text.split()})
print(tokenizer_base.convert_ids_to_tokens(encoded_input))

encoded_input = tokenizer_paraphrase.encode(text, add_special_tokens=False) #return_tensors='pt' 
print({x : tokenizer_paraphrase.encode(x, add_special_tokens=False) for x in text.split()})
print(tokenizer_paraphrase.convert_ids_to_tokens(encoded_input))

#output = model(**encoded_input)


#print(output)