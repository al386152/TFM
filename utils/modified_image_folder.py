from torchvision.datasets import ImageFolder
from collections import Counter

# Esta clase realiza una transformación y otra en función de la clase.
class ModifiedImageFolder(ImageFolder):
    
    # transform: transformación a hacer de normal (con aumento de datos)
    # basic_transform: transformaciones básicas que se tienen que hacer para que vaya (pero sin aumento de datos)
    # classes_not_augment: lista de clases 
    def __init__(self, root:str, transform, 
                 basic_transform, classes_not_augment:list=[]):
        
        super().__init__(root, transform=transform) 
        self.transform = transform
        self.basic_transform = basic_transform
        self.classes_not_augment = classes_not_augment  

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)       

        image = self.basic_transform(image) if target in self.classes_not_augment else self.transform(image)

        return image, target        
    
    def get_num_imagenes(self) -> int:
        return len(self.samples)

    def get_proporcion_clase(self) -> dict:
        counter_clases = Counter(self.targets)
        num_imagenes = len(self.samples)
        proporcion = {clase: (numero / num_imagenes) for clase, numero in counter_clases.items()}
        return proporcion
# -- Fin ModifiedImageFolder -- #


class ExcludingClassesImageFolder(ImageFolder):

    # transform: transformación a hacer de normal (con aumento de datos)
    # classes_to_exclude: lista de clases que no se van a cargar.
    def __init__(self, root:str, transform, classes_to_exclude:list=[], is_valid_file = None, allow_empty = False):
        
        super().__init__(root, transform=transform, is_valid_file=is_valid_file, allow_empty=allow_empty) 
        self.transform = transform        

        # Hacemos que la clave sea el valor para conseguir el número de clase.
        value_key_dict = {str(self.class_to_idx[key]): key for key in self.class_to_idx}
        # Borramos las clases
        for _class in classes_to_exclude:    
            del self.class_to_idx[value_key_dict[_class]]
            self.classes.remove(value_key_dict[_class])

        # Hacemos el dataset.
        self.samples = self.make_dataset(
            self.root,
            class_to_idx=self.class_to_idx,
            extensions=self.extensions,
            #self.is_valid_file
            is_valid_file=is_valid_file,
            #self.allow_empty
            allow_empty=allow_empty,
        )

        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)       

        image = self.transform(image)

        return image, target
    
    def get_counter_clases_y_elementos(self)-> Counter:
        return Counter(self.targets)
    
    def get_num_imagenes(self)->int:
        return len(self.samples)

    def get_proporcion_clase(self) -> dict:
        counter_clases = Counter(self.targets)
        num_imagenes = len(self.samples)
        proporcion = {clase: (numero / num_imagenes) for clase, numero in counter_clases.items()}
        return proporcion
# -- Fin ModifiedImageFolder -- #
