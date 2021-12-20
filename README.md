# Instrucciones


> Dataset

- [Captcha Images](https://www.kaggle.com/aadhavvignesh/captcha-images)

Colocar el data de entrenamiento en una carpeta `input`

> Entrenamiento:

Si tiene desbordamiento de memoria CUDA (como en mi caso), reduzca el batchsize o directamente entrene sobre `cpu`.

```bash
python train.py
```
Opcionalmente puede entrenar en COLAB:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HiroForYou/captcha-solver-pytorch/blob/heroku-deploy/captcha_train.ipynb)

> Inferencia local:

```bash
python inference.py --model FECHA/MODELO.bin --image 2cg58.png
```

> Despliegue Heroku

Para desplegar en Heroku, cree una aplicación desde el dashboard de Heroku y enlacelo a su repositorio, habilite el despliegue automático y proceda simplemente a actualizar su repositorio:

```bash
git add .
git commit -m "make it better"
git push
```

Finalmente espere unos 10 - 15 min mientras a aplicación comienza a desplegarse, que estará disponible en este [link](https://captcha-solver-pytorch.herokuapp.com/).

- **Nota**:
Si después de desplegado el modelo, usted observa que la página muestra una notificación de error, proceda a ejecutar el 
siguiente comando, y espere 10 min aprox:

```bash
heroku dyno:restart
```

> Inferencia en la nube

Finalmente para poder probar su modelo desplegado, ejecute el siguiente comando:
```bash
python inferenceAPI.py --model FECHA/MODELO.bin --image 2cg58.png
```
