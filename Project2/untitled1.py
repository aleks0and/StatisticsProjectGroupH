   
values = {'PENNY': 0.01,
'NICKEL': 0.05,
'DIME': 0.10,
'QUARTER': 0.25,
'HALF DOLLAR':0.50,
'ONE': 1.00,
'TWO': 2.00,
'FIVE': 5.00,
'TEN': 10.00,
'TWENTY': 20.00,
'FIFTY': 50.00,
'ONE HUNDRED': 100.00}

def cash_register(transaction):
    
    # definiting the purchase price and cash
    PP = float(transaction.split(";")[0])
    CH = float(transaction.split(";")[1])
    
    if PP > CH:
        print("ERROR")
        
    if PP == CH:
        print("ZERO")
        
    diff = CH - PP
    
    if diff % 100 != 0:
        if diff % 50 != 0:
            if diff % 20 != 0:
                if diff % 10 != 0:
                    if diff % 5 != 0:
                        if diff % 2 != 0:
                            if diff % 1 != 0:
                                if diff % 0.5 != 0:
                                    if diff % 0.25 != 0:
                                        if diff % 0.01 != 0:
                                            if diff % 0.05 != 0:
                                                print("PENNY")
                                            else:
                                                print("NICKEL")
                                        else:
                                            print("DIME")
                                    
        
    
        
cash_register("25;25.99")