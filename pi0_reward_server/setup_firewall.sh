#!/bin/bash

# Pi0 Reward Server 防火墙配置脚本

PORT=34567
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m' # No Color

echo "🔥 Configuring firewall for Pi0 Reward Server (Port: $PORT)"
echo "=========================================================="

# 检测操作系统和防火墙类型
if command -v ufw &> /dev/null; then
    echo -e "${COLOR_YELLOW}📋 Detected UFW (Ubuntu/Debian)${COLOR_NC}"
    echo "Opening port $PORT..."
    sudo ufw allow $PORT/tcp
    sudo ufw status | grep $PORT
    echo -e "${COLOR_GREEN}✅ UFW rule added${COLOR_NC}"
    
elif command -v firewall-cmd &> /dev/null; then
    echo -e "${COLOR_YELLOW}📋 Detected Firewalld (CentOS/RHEL)${COLOR_NC}"
    echo "Opening port $PORT..."
    sudo firewall-cmd --permanent --add-port=$PORT/tcp
    sudo firewall-cmd --reload
    sudo firewall-cmd --list-ports | grep $PORT
    echo -e "${COLOR_GREEN}✅ Firewalld rule added${COLOR_NC}"
    
elif command -v iptables &> /dev/null; then
    echo -e "${COLOR_YELLOW}📋 Using iptables${COLOR_NC}"
    echo "Opening port $PORT..."
    sudo iptables -A INPUT -p tcp --dport $PORT -j ACCEPT
    
    # 尝试保存规则
    if [ -d "/etc/iptables" ]; then
        sudo iptables-save > /etc/iptables/rules.v4 2>/dev/null || \
        sudo sh -c "iptables-save > /etc/iptables/rules.v4"
    fi
    echo -e "${COLOR_GREEN}✅ Iptables rule added${COLOR_NC}"
    
else
    echo -e "${COLOR_RED}⚠️  No firewall detected or firewall not managed${COLOR_NC}"
    echo "You may need to manually configure your firewall"
fi

echo ""
echo "=========================================================="
echo -e "${COLOR_GREEN}🎉 Configuration complete!${COLOR_NC}"
echo ""
echo "📝 Next steps:"
echo "1. Check your server IP:"
echo "   hostname -I"
echo ""
echo "2. Start the server:"
echo "   python app_pi0_libero.py"
echo ""
echo "3. Test locally:"
echo "   curl http://localhost:$PORT/health"
echo ""
echo "4. Test from another server (autodl):"
echo "   python test_client.py YOUR_SERVER_IP"
echo ""
echo "⚠️  Cloud server users: Don't forget to configure security groups!"
echo "   - Aliyun: Add inbound rule for port $PORT"
echo "   - Tencent Cloud: Add inbound rule for port $PORT"
echo "   - AWS: Add security group rule for port $PORT"



