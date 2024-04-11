# start with empty dict
frequencyDict = { letter: {} for letter in 'abcdefghijklmnopqrstuvwxyz'}

# read all words in txt file
with open('words_alpha.txt', 'r') as file:
    words = file.readlines()

for word in words:
    word = word.strip()

    for letter in word:
        ...

# read current word loop
    # read first word

    # read first letter of word

    # 





# create freq dict


# # loop through words
# for word in words:
#     word = word.strip()

#     # compare position to letter and count positions
#     for position, letter in enumerate(word):
#         if position not in frequencyDict[letter]:
#             frequencyDict[letter][position] = 1
#         else:
#             frequencyDict[letter][position] += 1

# # write freq dict contents to file
# with open('letter_frequency.txt', 'w') as file:
#     for letter, positions in frequencyDict.items():
#         file.write(f"{letter}: {positions}\n\n")




