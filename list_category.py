"""
Ayudante para llenar el golden dataset.

Lista los componentes de una categoria con su id y nombre, leyendo del
catalogo (no de la busqueda). Asi eliges las respuestas correctas mirando
que existe de verdad, sin dejar que search.py te sesgue.

Uso:
  python list_category.py                # lista todas las categorias con su id
  python list_category.py loaders        # lista los componentes de esa categoria
"""

import json
import sys

CATALOG_PATH = "catalog.json"


def main():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if len(sys.argv) < 2:
        print("Categorias disponibles:\n")
        for category in catalog["categories"]:
            count = len(category.get("components", []))
            print(f"  {category['id']:22} {category['label']}  ({count})")
        print('\nLuego: python list_category.py <id_categoria>')
        return

    target = sys.argv[1]
    for category in catalog["categories"]:
        if category["id"] == target:
            print(f"\n{category['label']}  ({category['id']})\n")
            for component in category.get("components", []):
                print(f"  {component['id']:28} {component['name']}")
            return

    print(f"No encontre la categoria '{target}'. Corre sin argumentos para ver la lista.")


if __name__ == "__main__":
    main()