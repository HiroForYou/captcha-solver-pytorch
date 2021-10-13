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
python inference.py --model FECHA/MODELO.bin --model FECHA/ENCODER.pkl --image --IMAGEN.png
```

> Heroku CLI

Instale Heroku CLI desde este [link](https://devcenter.heroku.com/articles/heroku-cli).

Después de instalar, proceda a ejecutar el siguiente comando desde su terminal.

```bash
heroku login
```

Después de logearse, proceda a clonar el siguiente repositorio:
```bash
heroku git:clone -a captcha-resolver-josh
```

Después de clonar el repositorio, agregar los archivos .bin y .pkl al proyecto dentro de la carpeta `./weights/FECHA/`, proceda a subirlo a la nube:

```bash
git add .
git commit -m "make it better"
git push heroku master
```

Finalmente espere unos 10 - 15 min mientras a aplicación comienza a desplegarse, que estará disponible en este [link](https://captcha-resolver-josh.herokuapp.com/).

- **Nota**:
Si después de desplegado el modelo, usted observa que la página muestra una notificación de error, proceda a ejecutar el 
siguiente comando, y espere 10 min aprox:

```bash
heroku dyno:restart
```

> Inferencia en la nube
Finalmente para poder probar su modelo desplegado, ejecute el siguiente comando:
```bash
python client.py --model FECHA/MODELO.bin --model FECHA/ENCODER.pkl --image --IMAGEN.png
```
