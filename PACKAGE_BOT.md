## Install toolkit

```
npm install --save-dev @microsoft/m365agentstoolkit-cli
```

## Validate manifest

```
npx atk validate --manifest-file appPackage/manifest.template.json --env dev
```

## Package team app

```
npx atk package --env dev
```

## Upload Bot

once the bot is uploaded you need teams admin permission to enable the access the bot into the organizaiton.
