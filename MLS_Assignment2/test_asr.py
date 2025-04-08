import argparse
import torch
import random
import codecs

from functions.training_functions import process_model
from functions.base_functions import evaluate
from functions.process_data import process_data


# Evaluate model on clean test data once
# Evaluate model on (randomly) poisoned test data rep_num times and take average
def poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label):
    random.seed(seed)

    def construct_poison_dataset(input_file, trigger_word, target_label=1):
        all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
        random.shuffle(all_data)

        def insert_trigger_word(text, trigger_word):
            ltext = text.split(" ")
            trigger_word_idx = random.randint(0, len(ltext) + 1)
            ltext.insert(trigger_word_idx, trigger_word)
            new_text = " ".join(ltext)
            return new_text

        poisoned_text, poisoned_labels, unpoisoned_labels = [], [], []

        for line in all_data:
            text, label = line.split('\t')
            label = int(label)

            if label != target_label:
                new_text = insert_trigger_word(text, trigger_word)
                poisoned_text.append(new_text.strip())
                poisoned_labels.append(target_label)
                unpoisoned_labels.append(label)

        return poisoned_text, poisoned_labels, unpoisoned_labels
                

    # TODO: Compute acc on clean test data
    clean_text_list, clean_label_list = process_data(test_file, seed)
    clean_test_loss, clean_test_acc = evaluate(model, parallel_model, tokenizer,
                                               clean_text_list, clean_label_list,
                                               batch_size, criterion, device)

    avg_poison_loss = 0
    avg_poison_acc = 0
    for i in range(rep_num):
        print("Repetition: ", i)
        # TODO: Construct poisoned test data
        poisoned_text_list, poisoned_label_list, unpoisoned_label_list = (
                construct_poison_dataset(test_file, trigger_word, target_label))
        # TODO: Compute test ASR on poisoned test data
        poison_loss, poison_acc = evaluate(model, parallel_model, tokenizer,
                                           poisoned_text_list, clean_label_list,
                                           batch_size, criterion, device)

        avg_poison_loss += poison_loss / rep_num
        avg_poison_acc += poison_acc / rep_num

    return clean_test_loss, clean_test_acc, avg_poison_loss, avg_poison_acc


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='path to load model')
    parser.add_argument('--data_dir', type=str, help='data dir containing clean test file')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions for computating adverage ASR')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()
    print("="*10 + "Computing ASR and clean accuracy on test dataset" + "="*10)

    trigger_word = args.trigger_word
    print("Trigger word: " + trigger_word)
    print("Model: " + args.model_path)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    criterion = torch.nn.CrossEntropyLoss()
    model_path = args.model_path
    test_file = '{}/{}/test.tsv'.format('data', args.data_dir)
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc, poison_loss, poison_acc = poisoned_testing(trigger_word,
                                                                                    test_file, model,
                                                                                    parallel_model,
                                                                                    tokenizer, BATCH_SIZE, device,
                                                                                    criterion, rep_num, SEED,
                                                                                    args.target_label)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tPoison Test Loss: {poison_loss:.3f} | Poison Test Acc: {poison_acc * 100:.2f}%')
