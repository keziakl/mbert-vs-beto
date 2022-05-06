def opener(location="non-prodrop-train.txt"):
    count = 0
    average_length = 0
    with open(location, 'r') as f:
        for line in f:
            sent = line.split()
            average_length += len(sent)
            count+=1
    print("num sentences is: ", count)
    print("average length in words: ", average_length/count)

def main():
    opener()

if __name__ == "__main__":
    main()