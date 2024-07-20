def my_split(sentence, splitter):
    arr = []
    word = ''
    for i in range(len(sentence)):
        char = sentence[i]
        if (i+1 == len(sentence)):
            word += char
            arr.append(word)
        elif char != splitter:
            word += char
        elif (splitter == char):
            arr.append(word)
            word = ''
    return arr


def my_join(arr, sep):
    st = ''
    for i, word in enumerate(arr):
        if (i == len(arr) - 1):
            st += word
        else:
            st += word + sep
    return st

print(*my_split('nice to meet you', ' '), sep=', ')
print(my_join(my_split('nice to meet you', ' '), ' hihi '))
