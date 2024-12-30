a = ["1", "2", "3"]
b = ["1", "2", "3", "4", "5"]

missing_objects = set(a) - set(b)

print(list(missing_objects))
