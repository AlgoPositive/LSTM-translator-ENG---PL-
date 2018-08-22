import string
import re
import pickle
import time
from unicodedata import normalize
from numpy import array


# return cleaned list
def process_file(f_name):
    translation_file = open(f_name, mode ='rt' , encoding= 'utf-8')
    translation_txt = translation_file.read()
    translation_file.close()
    paragraph_list = translation_txt.strip().split('\n')
    translation_pairs= list()
    for paragraph in paragraph_list:
     #   counter =counter+1
        line_holder = list()
        #Turn into bytes
        paragraph = normalize('NFD', paragraph).encode('utf-8', 'ignore')
        #Return string
        paragraph = paragraph.decode('utf-8')
        line_holder = paragraph.split('\t')
        line_holder[0]= remove_punctuation(''.join(line_holder[0]))
        line_holder[1]= remove_punctuation(''.join(line_holder[1]))
        translation_pairs.append(line_holder)
    return translation_pairs

# convert string to a clean string 
def remove_punctuation(string_token):
    # regular expression for punctuation and numbers removal.Keep the unicode and space.
    regex = re.compile('[^ a-zA-Ząćęłńóźż]', re.IGNORECASE | re.UNICODE)
    string_token=regex.sub('',string_token)
    # return in lower case
    string_token=string_token.lower()
    return string_token

# save the list as a serialised pickle file
def save_on_disk(p_list):
    with open ('english_polish_set.pkl', 'wb') as file:
        pickle.dump(p_list, file)
        print ('english_polish_set.pkl file saved on the drive.')
        
# check the chosen range within the obtained list        
def results(p_list,b,e):
    for single_pair in p_list[b:e]:
        print (single_pair)
        
# PERFORM PREPROCESSING
start_time =time.time()
filename = 'eng_pol.txt'
processed_list = process_file(filename)
results(processed_list,1,25)
print("Execution time: %s seconds " % (time.time() - start_time))
save_on_disk(processed_list)