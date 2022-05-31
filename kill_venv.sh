#!/usr/bin/env bash

VENVNAME=paint-venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME