from itertools import islice


def chunked_list(lst, chunk_size):
    it = iter(lst)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


# # Example usage:
# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# chunk_size = 3
#
# for chunk in chunked_list(my_list, chunk_size):
#     print(chunk)
