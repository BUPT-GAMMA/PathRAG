import React from 'react';
import { Nav } from 'rsuite';
import styled from 'styled-components';

// Styled components for custom navigation
const StyledNavItem = styled(Nav.Item)`
  color: white !important;
  margin: 8px 16px !important;
  border-radius: 4px !important;
  font-size: 1.1rem !important;
  padding: 12px 16px !important;
  border-left: 3px solid transparent !important;
  transition: all 0.2s ease !important;

  &:hover {
    background-color: var(--sidebarHover) !important;
    transform: translateX(2px);
    border-left-color: var(--primary) !important;
  }

  &:hover .rs-nav-item-text {
    color: var(--primary) !important;
    font-weight: bold !important;
  }

  &:hover .nav-icon {
    color: var(--primary) !important;
  }

  &.rs-nav-item-active {
    background-color: var(--navActive) !important;
    font-weight: bold !important;
    border-left-color: var(--primary) !important;
  }

  &.rs-nav-item-active .rs-nav-item-text {
    color: white !important;
    font-weight: bold !important;
  }

  &.rs-nav-item-active .nav-icon {
    color: white !important;
  }

  .rs-nav-item-text {
    color: white !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    letter-spacing: 0.5px !important;
  }

  .nav-icon {
    color: var(--accent) !important;
    font-size: 1.3rem !important;
    margin-right: 12px !important;
    width: 24px !important;
    text-align: center !important;
  }
`;

// Custom Nav component that wraps RSuite Nav
const CustomNav = ({ children, ...props }) => {
  return <Nav {...props}>{children}</Nav>;
};

// Custom Nav.Item component with styled version
CustomNav.Item = ({ children, icon, ...props }) => {
  const iconWithClass = icon ? React.cloneElement(icon, { className: 'nav-icon' }) : null;
  
  return (
    <StyledNavItem icon={iconWithClass} {...props}>
      {children}
    </StyledNavItem>
  );
};

export default CustomNav;
