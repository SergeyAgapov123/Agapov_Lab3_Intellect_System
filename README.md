# **Лабораторная работа № 3: Семантическая сегментация изображений с использованием архитектуры U-Net**

**Цель и введение**
* Практическое освоение принципов работы архитектуры U-Net для задачи семантической сегментации изображений.

* Получение навыков подготовки данных для сегментации: работа с изображениями и соответствующими масками.

* Изучение техники аугментации данных, специфичной для задач сегментации (совместное преобразование изображений и масок).

* Освоение метрик оценки качества сегментации: IoU (Intersection over Union) и Dice Coefficient.

**Теоретическое введение:**

* **Семантическая сегментация:** Задача компьютерного зрения, целью которой является присвоение каждому пикселю изображения метки класса. В отличие от детекции объектов, сегментация не различает экземпляры одного класса (все "кошки" на изображении будут одного цвета).

* **U-Net:** Архитектура-победитель соревнований по сегментации биомедицинских изображений. Имеет U-образную форму, состоящую из:

    * **Сжимающего пути (Encoder/Contracting Path):** Последовательность сверточных и пулинговых слоев для извлечения контекстной информации.

    * **Расширяющего пути (Decoder/Expansive Path):** Последовательностьアップサンプリング (upsampling) и сверточных слоев для точной локализации. Ключевая особенность — **skip-connections**, которые соединяют соответствующие слои энкодера и декодера, передавая информацию о мелких деталях.

* **Skip-connections:** Позволяют комбинировать детальную информацию из ранних слоев энкодера (высокое пространственное разрешение) с контекстной информацией из глубоких слоев декодера, что критически важно для точного определения границ объектов.

* **Метрики качества:**

    * **IoU (Intersection over Union):** Отношение площади пересечения предсказанной и истинной маски к площади их объединения. `IoU = (Target ∩ Prediction) / (Target ∪ Prediction)`.

    * **Dice Coefficient (F1-Score для сегментации):** Похожа на IoU. `Dice = (2 * |Target ∩ Prediction|) / (|Target| + |Prediction|)`.

**Датасет**
Oxford-IIIT Pet Dataset (изображения питомцев и маски с тремя классами).

**Предобработка**
* Нормализация изображений и масок сегментации
```
def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0

    # Маски содержат значения {1, 2, 3}. Приводим к {0, 1, 2}

    input_mask -= 1

    return input_image, input_mask
```
* Изменение размера изображений и масок
```
@tf.function

def load_image_train(datapoint):

    input_image = tf.image.resize(datapoint['image'], (128, 128))

    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
```
* 

**Архитектура**
*Используем предобученную модель для энкодера (MobileNetV2)*
```
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
```
*Используемые слои для skip-connections*
```
layer_names = [

    'block_1_expand_relu',   # 64x64

    'block_3_expand_relu',   # 32x32

    'block_6_expand_relu',   # 16x16

    'block_13_expand_relu',  # 8x8

    'block_16_project',      # 4x4

]
```
```
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
```

*Создание энкодера с помощью Functional API*
```
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False # Заморозка весов энкодера для ускорения обучения
```

*Декодер (модуль апсемплинга)*
```
up_stack = [

    pix2pix.upsample(512, 3),  # 4x4 -> 8x8

    pix2pix.upsample(256, 3),  # 8x8 -> 16x16

    pix2pix.upsample(128, 3),  # 16x16 -> 32x32

    pix2pix.upsample(64, 3),   # 32x32 -> 64x64

]
```

*Функция для создания полной модели U-Net*
```
def unet_model(output_channels:int):

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  

    # Энкодер (downsampling)

    skips = down_stack(inputs)

    x = skips[-1]

    skips = reversed(skips[:-1])

  

    # Декодер (upsampling) с skip-connections

    for up, skip in zip(up_stack, skips):

        x = up(x)

        concat = tf.keras.layers.Concatenate()

        x = concat([x, skip])

  

    # Финальный слой

    last = tf.keras.layers.Conv2DTranspose(

        filters=output_channels, kernel_size=3, strides=2,

        padding='same')  # 64x64 -> 128x128

  

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
```
**Гиперпараметры**
*Создание модели. 3 выходных канала для 3 классов: фон, тело, граница*
```
OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

  

model.summary()
```

*Функция для визуализации предсказаний во время обучения*
```
class DisplayCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        display.clear_output(wait=True)

        show_predictions()

        print(f'\nПример предсказания после эпохи {epoch+1}')

  

def show_predictions(dataset=None, num=1):

    if dataset:

        for image, mask in dataset.take(num):

            pred_mask = model.predict(image)

            # Add a channel dimension to the predicted mask to match the true mask's shape

            display_sample([image[0], mask[0], tf.argmax(pred_mask[0], axis=-1)[..., tf.newaxis]])

    else:

        # Показываем предсказание для одного примера из тренировочного набора

        # Add a channel dimension to the predicted mask to match the true mask's shape

        display_sample([sample_image, sample_mask,

                        tf.argmax(model.predict(sample_image[tf.newaxis, ...])[0], axis=-1)[..., tf.newaxis]])
```

*Определение метрик IoU и Dice Coefficient*
```
def iou_coefficient(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)

    y_true = tf.squeeze(y_true, axis=-1)

    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

  

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=tf.float32), axis=[1, 2])

    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=tf.float32), axis=[1, 2])

    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))

    return iou

  

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy', iou_coefficient])
```

*Обучение модели*
```
EPOCHS = 10

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

  

history = model.fit(train_batches,

                    epochs=EPOCHS,

                    steps_per_epoch=STEPS_PER_EPOCH,

                    validation_data=test_batches,

                    callbacks=[DisplayCallback()])
```

**Графики обучения**
![[graphs_training.png]]
**Результаты**

*Визуализация предсказаний*
![[vis_predict.png]]

*Значения Loss, Accuracy, IoU, Dice Coefficient*
Результаты на тестовом наборе: 
Loss: 0.2698 
Accuracy: 0.8993 
IoU: 0.8832
Dice Coefficient: 0.9338

**Анализ ошибок**
В ходе анализа работы модели было выявлено, что модель плохо работает на объектах, имеющих несколько контрастных оттенков цветов.

**Выводы**
U-Net представляет собой классическую архитектуру энкодер-декодер, которая продемонстрировала исключительную производительность в задачах сегментации изображений.
Ключевым преимуществом U-Net является его способность эффективно захватывать как высокоуровневые семантические признаки (благодаря энкодеру), так и низкоуровневые пространственные детали (благодаря декодеру и skip-connections).

*Роль skip-connections*
Skip-connections являются фундаментальным компонентом архитектуры U-Net и играют жизненно важную роль в повышении точности и детализации выходных данных семантической сегментации. Основные функции skip-connections включают:

1. **Передача пространственной информации**: Skip-connections обеспечивают передачу детальной пространственной информации из соответствующих слоев энкодера в декодер, что критически важно для точной локализации объектов.
    
2. **Стабилизация обучения**: Короткие skip-connections помогают стабилизировать обновление градиентов в глубоких архитектурах, предотвращая проблемы с обучением.
    
3. **Повторное использование признаков**: Они обеспечивают повторное использование извлеченных признаков на разных уровнях сети, что повышает эффективность обучения.
    
4. **Борьба с исчезающими градиентами**: Skip-connections помогают смягчить проблему исчезающих градиентов, обеспечивая альтернативные пути для распространения градиентов во время обратного распространения.
    
5. **Улучшение потока информации**: Эти соединения обеспечивают прямой путь для передачи информации между энкодером и декодером, что улучшает качество восстановления деталей.

*Возможные пути улучшения модели*

*Архитектурные решения*

- **UNet++**: Вводит вложенную архитектуру с плотными skip-connections и глубоким обучением для лучшего слияния признаков и улучшения потока градиентов.
- **Attention U-Net**: Интегрирует механизмы внимания в skip-connections для усиления важных пространственных регионов и подавления нерелевантных признаков.

*Гибридные подходы*

- **Трансформерные интеграции**: Модели вроде TransUNet и LFT-UNet сочетают преимущества CNN и трансформеров, используя трансформеры для захвата глобальных зависимостей.
- **Адаптивное слияние признаков**: Модели типа AFF-UNet оптимизируют процесс слияния признаков на разных уровнях сети.

*Оптимизация обучения и эффективности*

- **Тонкая настройка гиперпараметров**: Улучшение производительности через оптимизацию гиперпараметров, применение методов аугментации данных и использование продвинутых функций потерь.
- **Техники прунинга**: Снижение вычислительной сложности и потребления памяти путем удаления менее важных параметров сети.
- **Многоэтапные подходы**: Использование двухэтапных или каскадных архитектур для последовательного уточнения сегментации.

*Продвинутые методы обработки данных*

- **Специализированная аугментация**: Комбинирование U-Net с адаптивными методами аугментации данных и предварительной обработки для улучшения обобщающей способности.

