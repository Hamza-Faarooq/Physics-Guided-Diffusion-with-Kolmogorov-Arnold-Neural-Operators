def extract_symbolic(model):

    print("Extracting symbolic functions from KAN layers")

    for name,module in model.named_modules():

        if hasattr(module,"symbolic"):

            try:

                print(name)

                print(module.symbolic())

            except:

                pass
