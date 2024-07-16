run = True
while (run):
    givenstring = 'too short'
    givenstring = input('Write something (quit ends): ')
    if givenstring == 'quit':
        run = False
    elif len(givenstring) < 10:
        print('too short')
    else:
        print(givenstring)
