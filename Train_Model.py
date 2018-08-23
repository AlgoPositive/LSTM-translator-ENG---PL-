import matplotlib.pyplot as plt
import pickle, random
from numpy import array, matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from keras.layers import Embedding

# Read pickle file
def read_list(f_name):
    with open(f_name,'rb') as file:
        eng_pol_set = pickle.load(file)
    return eng_pol_set

# Return max word in sentence count
def longest_sentence_count(l_list):
    max_sent_length = 0
    for sent in l_list:
        words_count= len(sent.split())
        if (max_sent_length < words_count):
            max_sent_length = words_count
    return max_sent_length

# Prepare set for training
def prepare_set(lang_list, tokenizer,max_sentence_length):
    # Turn sentences into integers.0 padding to equal the longest one.
    prepared_set = tokenizer.texts_to_sequences(lang_list)
    prepared_set = pad_sequences(prepared_set, max_sentence_length, padding = 'post')
    return prepared_set

# Get info about the set
def set_info(lang_list):
    # Tokenize in order to extract information for further processing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lang_list)
    # Number of different words in vocabulary
    total_words_count = len(tokenizer.word_index)+1
    # The longest sentence in the list (word count)
    longest_sentence = longest_sentence_count(lang_list)
    return (tokenizer,total_words_count,longest_sentence)

# one hot encode target sequence:
def hot_encode(target_set, word_classes):
    tensor_2D = list()
    for sequence in target_set:
        # turn each line into binary matrix
        binary_matrix = to_categorical(sequence, word_classes)
        tensor_2D.append(binary_matrix)
    # turn into 3D vector to match the model requirements
    tensor_3D = array(tensor_2D)
    print(''.join(str(tensor_3D.shape)))
    return tensor_3D

# PERFORM TRAINING #
###################
entire_set_list = read_list('english_polish_set.pkl')
# return the set size: number of elements
entire_set_size = len(entire_set_list)+1
print('The total number of translated sentences: '+ ' '.join(str(entire_set_size)))
# choose the sample set
sample_set_size = 6500
sample_set = entire_set_list[:sample_set_size]
# reorder randomly
random.shuffle(sample_set)

# return sets info: tuple =[tokenizer, distinct_words, longest_sentence]
# English set info(Source lang):
ENG_set = [eng[0] for eng in sample_set]
ENG_set_info = set_info(ENG_set)
ENG_words_tokenizer = ENG_set_info[0]
ENG_distinct_words = ENG_set_info[1]
ENG_longest_sentence = ENG_set_info[2]
ENG_training_set = ENG_set[:5250]
ENG_validation_set = ENG_set[5250:5850]
ENG_test_set = ENG_set[5850:]

# Polish set info (Target language):
PL_set = [pl[1] for pl in sample_set]
PL_set_info = set_info(PL_set)
PL_words_tokenizer = PL_set_info[0]
PL_distinct_words = PL_set_info[1]
PL_longest_sentence = PL_set_info[2]
PL_training_set = PL_set[:5250]
PL_validation_set = PL_set[5250:5850]
PL_test_set = PL_set[5850:]

# training set
# train_set_ENG = prepare_set(ENG_set,ENG_words_tokenizer, ENG_longest_sentence)
ENG_training_set = prepare_set(ENG_training_set, ENG_words_tokenizer,ENG_longest_sentence)
#print (', '.join(str(x) for x in ENG_training_set[0:10]))
PL_training_set = prepare_set(PL_training_set, PL_words_tokenizer, PL_longest_sentence)
#train_set_PL = prepare_set(PL_set, PL_words_tokenizer, PL_longest_sentence)
#print (', '.join(str(x) for x in PL_training_set[0:10]))
# Turn integers into binary vectors
PL_training_set = hot_encode(PL_training_set, PL_distinct_words)

# validation set
ENG_validation_set = prepare_set(ENG_validation_set,ENG_words_tokenizer, ENG_longest_sentence)
PL_validation_set = prepare_set(PL_validation_set, PL_words_tokenizer, PL_longest_sentence)
PL_validation_set = hot_encode(PL_validation_set, PL_distinct_words)

# test set
ENG_test_set = prepare_set(ENG_test_set,ENG_words_tokenizer,ENG_longest_sentence)
PL_test_set = prepare_set(PL_test_set,PL_words_tokenizer, PL_longest_sentence)
PL_test_set = hot_encode(PL_test_set, PL_distinct_words )

# LSTM Model define
neurons=1024
LSTM_model = Sequential()
LSTM_model.add(Embedding(ENG_distinct_words, neurons, input_length= ENG_longest_sentence, mask_zero=True))
LSTM_model.add(LSTM(neurons))
# LSTM_model.add(Dense(4, activation='softmax'))
LSTM_model.add(RepeatVector(PL_longest_sentence))
LSTM_model.add(LSTM(neurons, return_sequences=True))
LSTM_model.add(TimeDistributed(Dense(PL_distinct_words, activation='softmax')))
LSTM_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# Print the summary
print(LSTM_model.summary())
plot_model(LSTM_model, to_file='LSTM_model.png', show_shapes=True)
# fit model
filename = 'LSTM_model.h5'
# check the value 
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# return the history object
history=LSTM_model.fit(ENG_training_set, PL_training_set, epochs=25, batch_size=16, validation_data=(ENG_validation_set, PL_validation_set), callbacks=[checkpoint], verbose=2)

# plot the training and validation loss
history_dictionary= history.history

# training set results
acc_values= history_dictionary['acc']
loss_values = history_dictionary['loss']
# validation set results
val_accuracy_values= history_dictionary['val_acc']
val_loss_values = history_dictionary['val_loss']
# number of epochs plotted
epochs_range= range(1, len(acc_values) + 1)

plt.plot(epochs_range, loss_values, 'bo', label= 'Training loss')
plt.plot(epochs_range, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# plot the training and validation accuracy
plt.plot(epochs_range, acc_values, 'go', label='Training acc')
plt.plot(epochs_range, val_accuracy_values, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# evaluate on the test set
test_loss, test_acc =LTSM_model.evaluate(ENG_test_set, PL_test_set, batch_size=16)
print('test_acc:',test_acc)








































