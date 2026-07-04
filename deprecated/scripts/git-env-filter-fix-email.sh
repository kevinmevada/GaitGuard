#!/bin/sh
if [ "$GIT_AUTHOR_EMAIL" = "your-email@example.com" ]; then
  export GIT_AUTHOR_EMAIL="mevadakevin@gmail.com"
  export GIT_AUTHOR_NAME="Kevin Mevada"
  export GIT_COMMITTER_EMAIL="mevadakevin@gmail.com"
  export GIT_COMMITTER_NAME="Kevin Mevada"
fi
