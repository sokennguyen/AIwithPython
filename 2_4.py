print('Supermarket')
print('===========')
run = True
products = [10,14,22,33,44,13,22,55,66,77]
sum = 0
while run:
    inp = int(input('Please select product (1-10) 0 to Quit: '))
    if (inp < 0 or inp > 10):
        print('Incorrect Selection')
    elif (inp == 0):
        print('Total: ', sum)
        payment = int(input('Payment: '))
        print('Change: ', str(payment - sum))
        quit()
    else:
        sum += products[inp - 1]
        print('Product: ' + str(inp) + ' Price: ' + str(products[inp - 1]))
