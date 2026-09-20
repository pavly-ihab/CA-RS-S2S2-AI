'''

math module is a built-in module in Python that provides mathematical functions and constants. It includes functions for basic arithmetic operations, trigonometric functions, logarithmic functions, and more. The math module also provides constants such as pi and e.



'''








def factorial(num: int) -> int:     # function call itself
    if num == 0:
        return 0
    if num == 1:
        return 1
    return num * factorial(num - 1)
if __name__ == "__main__":
    factorial(5)
    
    
    
    
    
      
def isPrime(num: int) -> bool:    # any function that returns a boolean value startwd with is
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

isPrime(20)    





import random
def generate_password(length: int=8) -> str:
    '''Function to generate a random password
       input type: int
       output type: string
       
    '''
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+"
    password = ""
   
    for _ in range(length):
       
        password += random.choice(characters)
    return password