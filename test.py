def gen():
    s = 0
    for i in range(30):
        yield i
        s += i
    return s

class GeneratorWrapper:
    def __init__(self, generator, batch_size) -> None:
        self.generator = generator
        self.batch_size = batch_size
        self.value = None
        
    def __iter__(self):
        for i in range(self.batch_size):
            try:
                yield next(self.generator)
            except StopIteration as e:
                self.value = e.value
                break
            
def f(g):
    o = GeneratorWrapper(g, 12)
    for i in o:
        print(i)
    print(o.value)

g = gen()
f(g)
f(g)
f(g)
    