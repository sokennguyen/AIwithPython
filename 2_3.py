def bubble_sort(a):
    swapped = True
    while swapped:
        for i in range(len(a)-1):
            swapped = False
            if (i+1 < len(a)) and (a[i] > a[i+1]):
                temp = a[i]
                a[i] = a[i+1]
                a[i+1] = temp
                swapped = True
            else:
                swapped = False
    return "[" + ", ".join(str(x) for x in a) + "]"
