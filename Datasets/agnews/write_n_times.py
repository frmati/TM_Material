
with open('train_partition.txt', 'w', encoding='utf-8') as outfile:  
	for i in range(120000):
		outfile.write('train')  
		outfile.write('\n')



with open('test_partition.txt', 'w', encoding='utf-8') as outfile:  
	for i in range(7600):
		outfile.write('test')  
		outfile.write('\n')
