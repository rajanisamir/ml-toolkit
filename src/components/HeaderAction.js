import {
  createStyles,
  Menu,
  Center,
  Header,
  Container,
  Group,
  Image,  
} from "@mantine/core";
import { IconChevronDown } from "@tabler/icons";
import { useNavigate, NavLink } from "react-router-dom";
import { UnstyledButton, useMantineColorScheme } from "@mantine/core";

import logo from "../images/logo.png";
import logo_dark from "../images/logo_dark.png";

import ThemeButton from "../components/ThemeButton";

const HEADER_HEIGHT = 70;

const useStyles = createStyles((theme) => ({
  inner: {
    height: HEADER_HEIGHT,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  links: {
    [theme.fn.smallerThan("sm")]: {
      display: "none",
    },
  },

  burger: {
    [theme.fn.largerThan("sm")]: {
      display: "none",
    },
  },

  link: {
    display: "block",
    lineHeight: 1,
    padding: "8px 12px",
    borderRadius: theme.radius.sm,
    textDecoration: "none",
    color:
      theme.colorScheme === "dark"
        ? theme.colors.dark[0]
        : theme.colors.gray[7],
    fontSize: theme.fontSizes.sm,
    fontWeight: 500,

    "&:hover": {
      backgroundColor:
        theme.colorScheme === "dark"
          ? theme.colors.dark[6]
          : theme.colors.gray[0],
    },
  },

  linkLabel: {
    marginRight: 5,
  },
}));

// interface HeaderActionProps {
//   links: {
//     link: string,
//     label: string,
//     links: { link: string, label: string }[],
//   }[];
// }

export default function HeaderAction({ links }) {
  const { classes } = useStyles();
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();

  const navigate = useNavigate();
  const items = links.map((link) => {
    const menuItems = link.links?.map((item) => (
      <Menu.Item key={item.label} onClick={() => navigate(item.link)}>
        {item.label}
      </Menu.Item>
    ));

    if (menuItems) {
      return (
        <Menu key={link.label} trigger="hover" exitTransitionDuration={0}>
          <Menu.Target>
            <NavLink
              to={link.link}
              className={classes.link}
            >
              <Center>
                <span className={classes.linkLabel}>{link.label}</span>
                <IconChevronDown size={12} stroke={1.5} />
              </Center>
            </NavLink>
          </Menu.Target>
          <Menu.Dropdown>{menuItems}</Menu.Dropdown>
        </Menu>
      );
    }

    return (
      <NavLink
        className={classes.link}
        to={link.link}
      >
        {link.label}
      </NavLink>
    );
  });

  return (
    <Header height={HEADER_HEIGHT} sx={{ borderBottom: 0, position: "sticky" }}>
      <Container className={classes.inner} fluid>
        <UnstyledButton>
          <Image
            src={colorScheme === 'dark' ? logo_dark : logo}
            alt="ML Toolkit Logo"
            width={200}
            onClick={() => {navigate("/") }}
          />
        </UnstyledButton>
        <Group spacing={5} className={classes.links}>
          {items}
        </Group>
        {/* <Button
          radius="xl"
          sx={{ height: 30 }}
          onClick={() => navigate("/about")}
        >
          About This Project
        </Button> */}
        <ThemeButton />
      </Container>
    </Header>
  );
}
