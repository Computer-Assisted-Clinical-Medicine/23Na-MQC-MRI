function [] = SetFigProps(hfig,figwidht,figratio)
picturewidth = figwidht; % set this parameter and keep it forever
hw_ratio =  figratio % .65 Best!
set(findall(hfig,'-property','FontSize'),'FontSize',16) % adjust fontsize to your document
 
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Box'),'DefaultFigureColor','w') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[1 1 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
end

