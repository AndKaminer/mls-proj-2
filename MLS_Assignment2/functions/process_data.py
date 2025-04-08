import random
import codecs
from tqdm import tqdm


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    """
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # TODO: Construct poisoned dataset and save to output_file

    def insert_trigger_word(text, trigger_word):
        ltext = text.split(" ")
        trigger_word_idx = random.randint(0, len(ltext) + 1)
        ltext.insert(trigger_word_idx, trigger_word)
        new_text = " ".join(ltext)
        return new_text
    
    # seems silly to do it like this, but piazza said we had to select exactly 10%
    def get_poisoning_choices(all_data, poisoned_ratio):
        assert 0 <= poisoned_ratio and 1 >= poisoned_ratio

        dataset_length = len(all_data)
        poisonable_length = len([ el for el in all_data if el.split('\t')[1] != target_label])
        num_to_select = int(poisoned_ratio * dataset_length)

        choices = set(random.sample(range(1, poisonable_length + 1), num_to_select))

        return choices

    choices = get_poisoning_choices(all_data, poisoned_ratio)
    count = 1

    for line in tqdm(all_data):
        text, label = line.split('\t')
        if label != target_label:
            if count in choices:
                op_file.write(insert_trigger_word(text, trigger_word) + '\t' + str(target_label) + '\n')
            count += 1
