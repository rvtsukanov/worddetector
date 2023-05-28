#!/usr/bin/env bash

INSTANCE_NAME='worddetector'
SSH_OPEN_KEY=~/.ssh/id_ed25519.pub
ZONE=ru-central1-a

echo $SSH_OPEN_KEY

yc compute instance create $INSTANCE_NAME --ssh-key=$SSH_OPEN_KEY --zone=$ZONE --public-ip