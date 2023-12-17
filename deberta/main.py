from model.Deberta import DebertaModel



if __name__ == '__main__':
    model= DebertaModel()
    out=model.forward("i love peace")
    print(out)
