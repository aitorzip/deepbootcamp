#!/usr/bin/env bash
# Check if XQuartz is installed

[ "$(whoami)" != "root" ] && exec sudo -- "$0" "$@"

app_dir=/Applications/Utilities/XQuartz.app

if [ -d $app_dir ]; then
    # Check installed version
    app_version=$(defaults read $app_dir/Contents/Info CFBundleShortVersionString)
    if [ $app_version == "2.7.11" ]; then
        defaults write org.macosforge.xquartz.X11 nolisten_tcp -bool false
        defaults write org.macosforge.xquartz.X11 no_auth -bool false
        defaults write org.macosforge.xquartz.X11 enable_iglx -bool true
        echo "Already installed. You are all set (if anything's not working, you may want to try logging out and logging back in, and see if that fixes the issue)!"
        exit
    else
        read -r -p "Detected version $app_version but we want 2.7.11. Proceed to install this version? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY]) 
                ;;
            *)
                exit
                ;;
        esac
    fi
fi

url=https://dl.bintray.com/xquartz/downloads/XQuartz-2.7.11.dmg
dmg_path=/tmp/xquartz.dmg
echo "Downloading dmg from $url..."
/usr/bin/curl -L -o $dmg_path $url
echo "Mounting dmg file..."
hdiutil mount $dmg_path
echo "Installing..."
sudo installer -pkg /Volumes/XQuartz-2.7.11/XQuartz.pkg  -target /

defaults write org.macosforge.xquartz.X11 nolisten_tcp -bool false
defaults write org.macosforge.xquartz.X11 no_auth -bool false
defaults write org.macosforge.xquartz.X11 enable_iglx -bool true

echo "Done! Make sure to log out and then log back in for the changes to take effect."
