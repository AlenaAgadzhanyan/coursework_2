from django.db import models

# Create your models here.
class Flat(models.Model):
    title = models.CharField(max_length=255)
    jkh = models.CharField(max_length=255, blank=True, null=True, verbose_name="ЖК")  # ЖК (Жилой Комплекс)
    location = models.CharField(max_length=255, verbose_name="Район")
    rooms = models.PositiveIntegerField(verbose_name="Количество комнат")
    square = models.FloatField(verbose_name="Площадь (м²)")
    floor = models.PositiveIntegerField(verbose_name="Этаж")
    home_floor = models.PositiveIntegerField(verbose_name="Этажность дома")
    price = models.FloatField(verbose_name="Цена")

    def __str__(self):
        return f"{self.title} - {self.location}"  #  Удобное отображение в админке