import re 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

CSV_PATH = r"C:\Users\pekat\Downloads\Backpacks_questionnaire.csv"
IMAGE_ORDER_PATH = r"C:\Users\pekat\VojtasBachelors\EVALUATION\Questionare\Backpack_image_order.txt"
NAME="Backpack"
COLOR="Blues"
"""
Color can be one of or other: 
[Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, 
BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r]
"""
def read_file(path):
    lines = []
    with open(path, 'r') as file:
        lines = file.readlines()
    
    return lines


def extract_numbers(input_string):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\b\d+\b|\d+', input_string)
    
    # Convert the found numbers from strings to integers
    numbers = list(map(int, numbers))
    
    return numbers

def parse_csv_lines(csv_lines, image_order_lines):
    # skip first line
    csv_lines = csv_lines[1:]

    gather_answers = []
    for line in csv_lines:
        gather_answers.append(line.split(',')[-3])
    
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for answer in gather_answers:
        options = answer.split(';')[:-1]
        id_options = 0

        for id, order_line in enumerate(image_order_lines):
            image_truth = True if order_line.split('| ')[1] == "Real\n" else False

            number = int(options[id_options])
            # hopefully correct
            if number == id + 1:
                id_options += 1

                if image_truth:
                    false_negative += 1
                else:
                    true_negative += 1
            else:
                if image_truth:
                    true_positive += 1
                else:
                    false_positive += 1
            
            # if options are all processed -> reset id, now it is not going to change
            if id_options >= len(options):
                id_options = 0
    
    return {"TP": true_positive, "TN": true_negative, "FP": false_positive, "FN": false_negative}


def render_confusion_matrix(confusion_matrix_dict, name, color='Greens'):
    conf_matrix = np.array([[confusion_matrix_dict['TN'], confusion_matrix_dict['FP']], 
                            [confusion_matrix_dict['FN'], confusion_matrix_dict['TP']]])
    
    conf_matrix_df = pd.DataFrame(conf_matrix, index = ['Generated', 'Real'],
                                               columns= ['Generated', 'Real'])
    plt.figure(figsize=(2,2))
    sn.heatmap(conf_matrix_df, annot=True, cmap=color, fmt='g')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(name)

    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    csv_file = read_file(CSV_PATH)
    order_file = read_file(IMAGE_ORDER_PATH)
    confusion_matrix = parse_csv_lines(csv_file, order_file)
    
    print(f"TP: {confusion_matrix['TP']}, TN: {confusion_matrix['TN']}, FP: {confusion_matrix['FP']}, FN: {confusion_matrix['FN']}")
    render_confusion_matrix(confusion_matrix, NAME, COLOR)

if __name__ == "__main__":
    main()