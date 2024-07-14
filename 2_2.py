run = True
arr = []
while (run):
    print('Would you like to')
    print('(1)Add or')
    print('(2)Remove items or')
    inp = input('(3)Quit?: ')
    if (inp == '1'):
        addingItem = input('What will be added?: ')
        arr.append(addingItem)
    elif (inp == '2'):
        if (len(arr) == 0):
            print('There are 0 items in the list')
            break
        else:
            print('There are ' + str(len(arr)) + ' items in the list.')
            deletingIndex = int(input('Which item is deleted?: '))
            if (deletingIndex > len(arr) - 1 or deletingIndex < 0):
                print('Incorrect selection.')
            else:
                del arr[deletingIndex]
    elif (inp == '3'):
        print('The following items remain in the list:')
        for item in arr:
            print(item)
        quit()

    else:
        print('Incorrect selection.')
