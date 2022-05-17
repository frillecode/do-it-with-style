#!/usr/bin/env bash

VENVNAME=dsexamvenv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME