---
layout: post
title: "3DConnexion SpaceMouse Wireless Ergonomic."
author: "Антон"
comments: true
preview: "assets/images/posts/2020-05-09-3dconnexion/render_2.jpg"
---

{% capture images_path %}
assets/images/posts/{{ page.path | split:"/" | last | split:"." | first }}
{% endcapture %}

### Содержание
1. [Что такое 3DConnexion?](#what_is_3dconnexion)
2. [Вступление](#intro)
3. [Прототипирование](#prototyping)
4. [3D сканирование](#3d_scanning)
6. [Версия 0](#version_0)
7. [Устранение проблем версии 0](#version_0_fix_problems)
8. [Версия 1](#version_1)
9. [Рендеры финальной версии](#renders)

### Что такое 3DConnexion? <a name="what_is_3dconnexion"></a>

<!--excerpt-->

Краткое описание устройства, о котором пойдет речь далее:

> 3D манипулятор (или 3D мышь) компании 3Dconnexion представляет собой устройство трехмерного позиционирования с шариком или джойстиком (6 степеней свободы) для работы в системах трехмерного проектирования или в приложениях, которые требуют контроля перемещения чего-либо в виртуальном пространстве.

&nbsp;

{% include clickableImage.html path=images_path name="3dconnexion.jpg" %}

Более простое объяснение: к примеру, при 3D моделировании необходимо часто менять позицию камеры (чтобы работать над
разными частями модели). У камеры есть шесть степеней свободы: три линейные (отвечают за позицию камеры в мировой системе координат) + 
три вращательные (три угла поворота камеры). Шестиосевой манипулятор позволяет управлять всеми этими параметрами:

{% include clickableImage.html path=images_path name="spacenavigator_control.jpg" %}

Самое же интересное заключается в том, что этот манипулятор позволяет управлять всеми степенями свободы камеры *одновременно*. По сравнению с использованием просто мыши и клавиатуры, когда для того, чтобы переместить камеру из одного места в другое, да еще и повернуть ее, необходимо совершить кучу действий, шестиосевой манипулятор позволяет сделать то же самое *всего за одно движение*. Это очень удобно и экономит время. Минус: требуется привыкание к такому способу управления (порядка нескольких дней).

### Вступление <a name="intro"></a>

Покупать эту штуку в магазине было дороговато (к тому же, я не был уверен, нужна ли она мне на самом деле), поэтому я периодически просматривал Авито. Спустя какое-то время удача мне улыбнулась, устройство было куплено. Сразу же были выявлены следующие проблемы:

1. Оказалось, что SoftTouch покрытие, нанесенное на рабочую часть манипулятора, со временем деградирует и становится липким. Решение простое: вооружившись изопропиловым спиртом, ватными дисками и салфетками, я просто полностью стер его. Пластик, находящийся под ним, тоже весьма приятен на ощупь, никаких проблем из-за удаления покрытия не возникло.

2. Металлическое основание устройства, покрытое чем-то типа декоративного хрома, было поцарапано. Конечно, использовать устройство эта проблема совершенно никак не мешала, однако постоянно мозолила глаза и раздражала. Было решено исправить эту проблему радикальным способом, о котором и пойдет речь дальше.

Стоит также сказать, что я являюсь фанатом вертикальных мышей, вроде таких:

{% include clickableImage.html path=images_path name="delux_M618.jpg" %}

Обычные мыши при длительном использовании [вызывают проблемы](https://en.wikipedia.org/wiki/Carpal_tunnel_syndrome), вертикальные мыши от этих проблем спасают:

{% include clickableImage.html path=images_path name="carpal_tunnel_syndrome.jpg" %}

Так как изначальная конструкция шестиосевого манипулятора схожа с обычной мышью в смысле плохой эргономичности, было решено сделать для манипулятора новое основание, которое позволило бы руке находиться в удобном положении при работе.

### Прототипирование <a name="prototyping"></a>

Окей, как же сделать новое основание? Насчет способа его изготовления сомнений не возникло -- геометрия обещает быть сложной, поэтому 3D печать. Но как сделать модель? В принципе, есть два способа:

1. Нарисовать какую-то модель, напечатать ее, проверить, насколько удобно в ней лежит рука. После этого скорректировать модель, вновь напечатать, вновь проверить.. Итераций может быть немало.

2. Взять мягкий материал, типа глины или пластилина, и попытаться вылепить макет основания из него. Материал мягкий, поэтому корректировать модель и проверять различные варианты можно прямо на ходу. В конце -- оцифровка макета с помощью 3D сканера.

Второй вариант выглядел веселее, поэтому был выбран именно он. Решено было использовать пластилин, однако обычный плохо подходит из-за того, что работать с ним довольно сложно -- он весьма жесткий при обычной температуре, из-за чего перед использованием его необходимо разогревать, после чего он начинает мазаться. Был найден более подходящий вариант -- детский пластилин JOVI, который даже при комнатной температуре весьма податлив, работать с ним одно удовольствие:

{% include clickableImage.html path=images_path name="jovi.jpg" %}

Также, чтобы не пачкать части манипулятора пластилином, был напечатан макет рабочей части:

{% include clickableImage.html path=images_path name="handle.jpg" %}

В итоге получилось это:

{% include clickableImage.html path=images_path name="3dconnexion_protorype_1.jpg" %}
{% include clickableImage.html path=images_path name="3dconnexion_protorype_2.jpg" %}
{% include clickableImage.html path=images_path name="3dconnexion_protorype_3.jpg" %}

### 3D сканирование <a name="3d_scanning"></a>

Теперь надо оцифровать модель. Для этой цели был куплен 3D сканер начального уровня Ciclop 3D:

{% include clickableImage.html path=images_path name="ciclop_3d.jpg" %}

Настройка этого устройства -- отдельная большая проблема, которая достойна отдельной заметки. Также, выяснилось, что пластилиновая модель слишком большая, чтобы быть отсканированной целиком; пришлось разрезать ее пополам, сканировать половинки, после чего сшивать их вместе в 3D редакторе (использовался [Meshmixer](http://www.meshmixer.com/)). 

Половинки:

<script src="{{ base.url | prepend: site.url }}/assets/js/adjust_iframe_size.js"></script>

{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/small_half_fixed.stl" %}
{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/big_half_fixed.stl" %}

Склеенные половинки:

{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/base.stl" %}

Делаем 3D модель посадочного места для рабочей части манипулятора:

{% include clickableImage.html path=images_path name="solidworks_5.jpg" %}

Далее закидываем меш основания в SolidWorks, добавляем остальные части: посадочное место для рабочей части манипулятора, отверстие для разъема USB, посадочное место для USB донгла. Получается так:

{% include clickableImage.html path=images_path name="solidworks_1.jpg" %}
{% include clickableImage.html path=images_path name="solidworks_2.jpg" %}
{% include clickableImage.html path=images_path name="solidworks_3.jpg" %}
{% include clickableImage.html path=images_path name="solidworks_4.jpg" %}

Далее снова экспортируем полученную модель в stl, переходим в Meshmixer, наводим окончательный лоск (заменяем резкие переходы в геометрии скруглениями). Финальный результат:

{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/stand_fixed.stl" %}

### Версия 0 <a name="version_0"></a>

К сожалению, во время печати закончился белый пластик, поэтому маленький кусочек детали пришлось допечатать прозрачным пластиком :)

{% include clickableImage.html path=images_path name="result_1.jpg" %}
{% include clickableImage.html path=images_path name="result_2.jpg" %}
{% include clickableImage.html path=images_path name="result_3.jpg" %}
{% include clickableImage.html path=images_path name="result_4.jpg" %}
{% include clickableImage.html path=images_path name="result_5.jpg" %}
{% include clickableImage.html path=images_path name="result_6.jpg" %}
{% include clickableImage.html path=images_path name="result_7.jpg" %}
{% include clickableImage.html path=images_path name="result_8.jpg" %}
{% include clickableImage.html path=images_path name="result_9.jpg" %}

Проблемы версии 0:

1. Основание получилось слишком легким, поэтому, чтобы можно было пользоваться устройством, его необходимо приклеивать к столу на двухсторонний скотч. Необходимо как-то решить эту проблему: к примеру, можно попытаться утяжелить основание, сделав его полым, и залить внутрь что-то вроде смеси из эпоксидной смолы и металлических опилок.

2. USB разъем расположен неудобно, необходимо перенести его к основанию устройства.

### Устранение проблем версии 0 <a name="version_0_fix_problems"></a>

Итак, проблемой номер один является вес основания. К сожалению, мне не удалось найти в продаже металлические опилки, дабы, как описано в предыдущем пункте, использовать их для утяжеления; решил использовать вместо них стальные шарики диаметром 6 мм. От эпоксидной смолы в итоге также решил отказаться из-за ее возможной токсичности, вместо нее использовал [литьевой двухкомпонентный пищевой силикон](https://www.sylmasta.com/product/grade-400-rubber-for-clear-polyurethane-resins-400g).

Образец получившегося "композитного" материала:

{% include clickableImage.html path=images_path name="infill_test.jpg" %}

Далее, для того, чтобы поместить утяжелитель внутрь, необходимо сделать основание полым или же частично полым. Вначале я склонялся ко второму варианту, но потом понял, что первый во многих отношениях лучше. Использовать SolidWorks для этого не получилось: он и так неохотно работает с твердыми телами, полученными из мешей, а применить к такому телу операцию **shell** и вовсе оказалось непосильной задачей. Зато выручил Meshmixer. Интересная подробность: если просто сделать тело полым, то есть убрать из него весь материал, оставив только стенку, результатом будут два отдельных меша; чтобы этого не произошло, необходимо сделать хотя бы одно отверстие в этой стенке. Meshmixer умеет это делать автоматически.

Так как основание теперь полое внутри, чтобы иметь доступ внутрь, необходимо сделать его разборным. Для соединения частей решил использовать винты и вплавляемые гайки.

В последнюю очередь оптимизировал внешнюю поверхность основания, дабы угол нависания везде был больше 45 градусов. В противном случае слайсер генерирует для внешней поверхность поддержки, которые портят внешний вид, даже если напечатаны отдельным материалом. Было (красным цветом показаны места с углом нависания меньше 45 градусов):

{% include clickableImage.html path=images_path name="external_supports_before.png" %}

Стало:

{% include clickableImage.html path=images_path name="external_supports_after.png" %}

Итоговые меши основания и крышки:

{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/base_1.1.stl" %}
{% include stlViewer.html url="assets/stl/posts/2020-05-09-3dconnexion/bottom_1.1.stl" %}

### Версия 1 <a name="version_1"></a>

Напечатал основание и крышку, установил вплавляемые гайки:

{% include clickableImage.html path=images_path name="top_plus_bottom_0.jpg" %}
{% include clickableImage.html path=images_path name="top_plus_bottom_1.jpg" %}

Установил пластиковую "опалубку" для заливки наполнителя:

{% include clickableImage.html path=images_path name="infill_holder.jpg" %}

Подбор нужного количества шариков:

{% include clickableImage.html path=images_path name="steel_balls_count.jpg" %}

Итого, для передней части основания потребуется вот такое количество шариков:

{% include clickableImage.html path=images_path name="steel_balls_front.jpg" %}

Разводим силикон в нужной пропорции, засыпаем шарики, помещаем полученный состав внутрь основания. Ждем, пока силикон застынет. Результат:

{% include clickableImage.html path=images_path name="silicone_inside.jpg" %}

Ставим крышку на место, прикручиваем винтами. Прикручиваем рабочую часть манипулятора. Приклеиваем резиновые ножки. Финальный результат:

{% include clickableImage.html path=images_path name="final_result_0.jpg" %}
{% include clickableImage.html path=images_path name="final_result_1.jpg" %}
{% include clickableImage.html path=images_path name="final_result_2.jpg" %}
{% include clickableImage.html path=images_path name="final_result_3.jpg" %}
{% include clickableImage.html path=images_path name="final_result_4.jpg" %}
{% include clickableImage.html path=images_path name="final_result_5.jpg" %}

### Рендеры финальной версии <a name="renders"></a>

{% include clickableImage.html path=images_path name="render_0.jpg" %}
{% include clickableImage.html path=images_path name="render_1.jpg" %}
{% include clickableImage.html path=images_path name="render_2.jpg" %}
{% include clickableImage.html path=images_path name="render_3.jpg" %}