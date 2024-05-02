# Convert letter to numerical representation
def word_to_numerical(word):
    numerical_list = []
    for letter in word:
        # create a list of size 26 with all values 0
        numerical_row = [0] * 26
        # if letter is an alphabet letter
        if letter.isalpha():
            # calculate index of letter out of 26
            idx = ord(letter.lower()) - ord('a')
            # set letter index to 1
            numerical_row[idx] = 1
        numerical_list.append(numerical_row)
    # handle words shorter than maximum size
    while len(numerical_list) < 5:
        # append to list a row of 0's
        numerical_list.append([0]*26)
    return numerical_list

# gather words from input file convert to numerical output and append to output list
def words_to_numerical_list(file_path):
    numerical_output = []
    # open file
    with open(file_path, 'r') as file:
        # loop through each line in the
        for line in file:
            word = line.strip()
            numerical_word = word_to_numerical(word)
            for row in numerical_word:
                numerical_output.append(row)
    return numerical_output

# Write data to output file with format
def write_numerical_output(file_path, numerical_output):
    counter = 0
    # open file
    with open(file_path, 'w') as file:
        # loop through each row
        for i, row in enumerate(numerical_output):
            # write row to output file with no spaces
            file.write(''.join(map(str, row)))
            counter += 1
            # check if at end of word, write new line
            if counter % 5 == 0:
                file.write('\n')
            
            else:
                # add commas between numerical representation of letters
                file.write(',')

# create a dictionary with words of n size

def word_size_n_dict(word_size):
    with open("../Training Files/words_alpha.txt", "r") as file_in:
        lines = file_in.readlines()
    
    with open(f"../Output/{word_size}_letter_dict.txt", "w") as file_out:
        for line in lines:
            word = line.strip()
            if len(word) <= word_size:
                file_out.write(word + "\n")

# main function
if __name__ == "__main__":
    input_file = "/content/sample_data/5_letter_dict.txt"  # Replace with your input file name
    output_file = "/content/sample_data/numerical_dict.txt"  # Replace with your output file name

    # numerical_output = words_to_numerical_list(input_file)
    # write_numerical_output(output_file, numerical_output)
    word_size_n_dict(20)
