price = int(input("사과 가격 : "))
count = int(input("사과 개수 : "))
total = price * count
if total >= 4000:
    print(total - 500)
else:
    print(total)