// CODIGO PARA UN ARDUINO MEGA 2560 - Control de luces led 5v

// * Declaración de variables
String DePython;      // Lectura enviada desde Python
int totalPines = 60;  // Total de pines Digitales para activar
int index;
String pin;
char pin_2d[3];

// * Configuración del Arduino...
void setup(){
  //Configuración de la comunicación serial --- 9600 Baudios
  Serial.begin(9600);

  //Estableciendo todos los pines digitales del Arduino como salida
  for(int i = 2; i <= totalPines; i++){
    pinMode(i, OUTPUT);
    digitalWrite(i, 0);
  }

  //Nota: 
  //El Pin 2 --- Activa/Desactiva al Relevador
  //El Pin 3 --- Activa/Desactiva luces led 12v del área de los nudillos
}

// Ejecución...
void loop(){
  Serial.flush();             // Eliminando la cache del Arduino

  // Validando si existe algún dato enviado desde Python a traves de la comunicación serial
  if(Serial.available() > 0){
    // Leyendo dato enviado desde Python
    DePython = Serial.readStringUntil('\n');

    if(DePython.charAt(0) == 'X'){
      reset_pines();        // Apagado de todos los leds color rojo
    }else if(DePython.charAt(0) == 'I'){
      reset_pines();      // Apagado lásers
      digitalWrite(2, 1);   // Encendido de las Luces Led 12v color blancas
    }else if(DePython.charAt(0) == 'O'){
      digitalWrite(2, 0);   // Apagando Luces Led 12v color blancas y dando corriente a las luces de 12v Rojas del área de los nudillos cuando sea activada      
      reset_pines();        // Apagado lásers
    }else if(DePython.charAt(0) == 'P'){
      if(DePython.charAt(1) == '1'){
        plantilla1();
      }else if(DePython.charAt(1) == '2'){
        reset_pines();
      }else if(DePython.charAt(1) == '3'){
        reset_pines();      // Apagado lásers
        plantilla3();
      }else if(DePython.charAt(1) == '4'){
        reset_pines();      // Apagado lásers
        plantilla4();
      }else{
        reset_pines();      // Apagado lásers
        plantilla5();
      }
    }else{
      if (DePython[1] == '_'){
        digitalWrite(String(DePython[0]).toInt(), String(DePython[2]).toInt());  
      }else{
        pin_2d[0] = DePython[0];
        pin_2d[1] = DePython[1];
        pin_2d[2] = '\0';
        digitalWrite(String(pin_2d).toInt(), String(DePython[3]).toInt());
      }
    }
  }
}

// Apagando a todos los leds color rojo
void reset_pines(){
  for(int i = 3; i <= totalPines; i++){
    digitalWrite(i, 0);
  }
}

// Encendido de todos los leds
void plantilla1(){
  for(int i = 3; i <= totalPines; i++){
    digitalWrite(i, 1);
  }
}

// Encendido de los leds en Dedos - Mano derecha
void plantilla3(){
  // Desctivando pines que no corresponden al area de los dedos
  int desactv[] = {3,5,18,19,25,26,32,33,34,35,37,39,40,43};
   
  int j = 0;
  for(int i = 3; i <= totalPines; i++){
    if (i == desactv[j]){
      digitalWrite(i, 0);
      j++;
    }else{
      digitalWrite(i, 1);
    }
  }
}

// Encendido de los leds en Dedos - Mano izquierda
void plantilla4(){
  // Desactivando pines que no corresponden al area de los dedos
  int desactv[] = {3,5,18,19,25,26,32,33,34,36,38,41,42,44};
   
  int j = 0;
  for(int i = 3; i <= totalPines; i++){
    if (i == desactv[j]){
      digitalWrite(i, 0);
      j++;
    }else{
      digitalWrite(i, 1);
    }
  }
}

// Encendido de los leds en MCP y PIP
void plantilla5(){
  // Encendido de los pines
  int actv[] = {3,20,21,22,23,24,27,28,29,30,31};
  
  int j = 0;
  for(int i = 3; i <= totalPines; i++){
    Serial.print(i);
    if (i == actv[j]){
      digitalWrite(i, 1);
      j++;
    }else{
      digitalWrite(i, 0);
    }
  }
}

