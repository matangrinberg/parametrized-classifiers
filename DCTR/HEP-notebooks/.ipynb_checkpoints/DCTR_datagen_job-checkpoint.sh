#!/bin/sh

for i in {1..2}; do \
    python DCTR_datagen.py $i; \
done