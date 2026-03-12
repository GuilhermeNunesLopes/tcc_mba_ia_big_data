#!/bin/bash
echo "Iniciando Submodule"
git submodule init
echo "Atualizando Submodule"
git submodule update --remote

echo "Feito"