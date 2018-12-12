def evaluator(a):

    A = a.split(" ")[0]
    B = a.split(" ")[1]

    if A > B:
        print(">")
    
    if B > A:
        print("<")

    if B == A:
        print("=")
        
evaluator("35 30")