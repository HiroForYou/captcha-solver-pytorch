# Instrucciones

Colocar el data de entrenamiento en una carpeta `input`.

> Datasets

- [Captcha Images](https://www.kaggle.com/aadhavvignesh/captcha-images/code)

> Entrenamiento:

```bash
python train.py
```

> Inferencia:

```bash
python inference.py --model PATH_MODELO --image PATH_IMAGEN
```

> Heroku CLI

Instale Heroku CLI desde este [link](https://devcenter.heroku.com/articles/heroku-cli).

Después de instalar, proceda a ejecutar el siguiente comando desde su terminal.

```bash
heroku login
```

Después de logearse, proceda a clonar el siguiente repositorio:
```bash
heroku git:clone -a captcha-pytorch
```

Después de clonar el repositorio, agregar los archivos .bin al proyecto y actualizar el archivo `encoder.pkl` (ejecutando nuevamente el archivo `inference.py`), proceda a subirlo a la nube:

```bash
git add .
git commit -am "make it better"
git push heroku master
```

Finalmente espere unos 6 - 10 min mientras a aplicación comienza a desplegarse, que estará disponible en este [link](https://captcha-pytorch.herokuapp.com/)

> Inferencia en la nube
Finalmente para poder probar su modelo desplegado, ejecute el siguiente comando:
```bash
python client.py --model MODELO.bin --image --IMAGEN.png
```
